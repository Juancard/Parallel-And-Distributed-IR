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
        token.setActive(true);
    }

    public void activate() {
        token.setActive(true);
    }

    public void release() {
        token.setActive(false);
    }
}
