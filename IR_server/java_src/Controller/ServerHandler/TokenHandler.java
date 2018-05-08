package Controller.ServerHandler;

import Common.IRProtocol;
import Controller.Token;

/**
 * Created by juan on 08/05/18.
 */
public class TokenHandler {
    private Token token;

    public TokenHandler(Token token){
        this.token = token;
        token.setActive(false);
    }

    public Object activate() {
        token.setActive(true);
        return IRProtocol.TOKEN_ACTIVATE_OK;
    }

    public Object release() {
        token.setActive(false);
        return IRProtocol.TOKEN_RELEASE_OK;
    }
}
